<template>
    <div class="page">
        <div class="menu">
            <div class="item">
                <div class="title">model</div>
                <el-select class="value" size="small" v-model="params.model" @change="update">
                    <el-option v-for="model in models" :key="model" :value='model'/>
                </el-select>
                </div>
            <div class="item">
                <div class="title">input</div>
                <el-select class="value" size="small" v-model="params.input" @change="params.target = images[params.input]">
                    <el-option v-for="image in Object.keys(images)" :key="images[image]" :value='image'/>
                </el-select>
            </div>
            <div class="item"><div class="title"></div><el-button icon='el-icon-refresh' type="primary" size="small" circle  @click="update"/></div>
        </div>
        <div class="network">
            <div class="layer">
                <div class="name">input</div>
                <img :src="params.input" width="64px"/>
            </div>
            
            <div class="layer" v-for="(layer, index) in res" :key="index">
                <div class="name">{{layer.name}}</div>
                <div class="filters" v-for="(filter, index) in layer.filters" :key="index">
                    <img :src="filter"/>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
export default {
    data() {
        return {
            models: [],
            images: {},
            res: [],
            params: {
                model: 'alexnet',
                input: '',
            }
        };
    },
    created() {
        this.config()
    },
    sockets: {
        models(data) {
            this.models = data
        },
        images(data) {
            this.images = data

            this.params.input = Object.keys(data)[0]
        },
        response_filters(data) {
            this.res = data
        }
    },
    methods: {
        config() {
            this.$socket.emit('get_models')
            this.$socket.emit('get_images')
        },
        update() {
            this.$socket.emit("filters", this.params);
        },
    }
};
</script>

<style rel="stylesheet/scss" lang="scss" scoped>
.page {
    .network {
        display: flex;

        .layer {
            display: flex;
            flex-flow: column;
            align-items: center;
            padding: 0 5px;

            .filters {
                width: 70px;
                display: flex;
                flex-flow: column;
                flex-wrap: wrap;
                align-items: center;

                img {
                    width: 32px;
                    padding: 5px 0;
                }
            }
        }
    }
}

</style>
