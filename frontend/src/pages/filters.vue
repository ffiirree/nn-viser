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
                <el-select class="value" size="small" v-model="params.input" @change="update">
                    <el-option value='static/images/cat_dog.png'/>
                    <el-option value='static/images/spider.png'/>
                    <el-option value='static/images/snake.jpg'/>
                </el-select>
            </div>
        </div>
        <div class="network">
            <div class="layer">
                <div class="name">input</div>
                <img :src="params.input" crossorigin='anonymous' width="64px"/>
            </div>
            
            <div class="layer" v-for="(layer, index) in res" :key="index">
                <div class="name">{{layer.name}}</div>
                <div class="filters" v-for="(filter, index) in layer.filters" :key="index">
                    <img :src="filter" crossorigin='anonymous'/>
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
            res: [],
            params: {
                model: 'alexnet',
                input: 'static/images/cat_dog.png',
            }
        };
    },
    created() {
        this.config()
        this.update()
    },
    sockets: {
        models(data) {
            this.models = data
        },
        response_filters(data) {
            this.res = data
        }
    },
    methods: {
        config() {
            this.$socket.emit('get_models')
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
